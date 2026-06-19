# EGFR Atomistic Frustration Pipeline — Proje Durum Raporu

**Tarih:** 19 Haziran 2026  
**Repo:** `/home/tugba/egfr_atomic_resolution`  
**Ortam:** `conda activate frustrato` (Python 3.10)

---

## 1. Ne Yapıyorduk: Bilimsel Hedef

### Referans Makale

**Chen et al. (2020), *Nature Communications* 11, 5944**  
*"Surveying biomolecular frustration at atomic resolution"*  
DOI: [10.1038/s41467-020-19560-9](https://doi.org/10.1038/s41467-020-19560-9)

Makale, protein enerji peyzajının "ne kadar optimize edildiğini" ölçen **atomistik frustrasyon** kavramını sunuyor. Bir amino asit çiftinin (i, j) kendi native ortamında ne kadar stabil olduğu, aynı konumdaki rastgele yerleştirilmiş amino asit çiftleriyle karşılaştırılarak hesaplanıyor.

### Paper'ın EGFR Analizi (Figure 5)

Makalede EGFR kinaz domaininin 4 inhibitör kompleksi için şu gözlem yapılmış:

> **Güçlü bağlanan inhibitörler → ligand etrafındaki kontaklar daha az frustrasyonlu**  
> Zayıf bağlayan inhibitörler → daha frustrasyonlu ligand-protein arayüzü

Paper bu analizi sadece **görselleştirme** amacıyla yapmış (4 yapı). Korelasyon analizi yapmamışlar ve orijinal kod **hiç yayınlanmamış**.

---

## 2. Bizim Katkımız: Üzerine Ne Koyduk

### Genişletme

| Boyut | Paper | Bu Proje |
|---|---|---|
| Yapı sayısı | 4 | **25** |
| Hedef | Görselleştirme | **Affinite-frustrasyon korelasyonu** |
| Kod | Yayınlanmamış | **Bağımsız reimplementasyon** |
| Affinite | Sadece Kd | Kd + IC50 (karma, log-dönüşüm) |
| Kovalan inhibitörler | Dahil | **Çıkarıldı** (3IKA, 4LQM, 5XDK) |

### Temel Matematiksel Reimplementasyon

**Denklem 1 — Frustrasyon indeksi (Z-skoru):**
```
F_ij = (E_ij_native − mean(E_ij_decoy)) / std(E_ij_decoy)
```
- `F > 0.78`  → **minimally frustrated** (native çok stabil)
- `F < −1.0`  → **highly frustrated**
- Arada       → neutral

**Denklem 2 — Many-body pairwise enerji düzeltmesi:**
```
E_ij = e_ij + 0.5 × Σ_{k∈contacts(i), k≠j} e_ik
             + 0.5 × Σ_{l∈contacts(j), l≠i} e_jl
```
- `e_ij`: doğrudan çift enerjisi (REF2015, `fa_rep` hariç)
- Arka plan kontakları her iki residue'nun komşularından yarı katkı alır

---

## 3. 25 EGFR–İnhibitör Kompleksi

### Affinite Dağılımı

| Özellik | Değer |
|---|---|
| Toplam yapı | 25 |
| Kd (termodinamik) | 8 |
| IC50 (enzim) | 17 |
| Affinite aralığı | 0.8 pM – 10 mM |
| log10(pM) aralığı | −0.10 – 7.00 (**~7 log birim**) |
| Kristal çözünürlüğü | 1.70 – 2.93 Å |
| Kaynak | 4 paper + 21 BindingDB |

### Özel Durumlar

- **7JXM (EAI045):** Allosterik inhibitör. Protein A/B/C/D zincirlerinde; ligand `9LL` yalnızca B ve D'de → `chain=B` override.
- **Çıkarılan yapılar:** 4LQM/DJK, 5XDK/8JC, 3IKA/0UN → kovalan inhibitörler, frustrasyon hesabı için uygun değil.

---

## 4. Yazılan Kod (~1754 satır)

### `src/frustration.py` (542 satır) — Çekirdek Motor

| Fonksiyon | Açıklama |
|-----------|----------|
| `get_protein_contacts` | Cα–Cα ≤ 10 Å kontak listesi |
| `get_ligand_contacts` | Ligand heavy-atom – Cα ≤ 10 Å |
| `pairwise_energy` | Direkt e_ij (REF2015, fa_rep hariç) |
| `contact_energy_eq2` | Tam Denklem 2 |
| `native_aa_frequency` | Native AA frekans dağılımı |
| `generate_decoy` | AA shuffle → side-chain repack → 1 decoy |
| `run_frustration_survey` | N decoy Z-skoru → DataFrame |
| `summarize_ligand_frustration` | Ligand arayüzü özeti |

### `src/prepare_structures.py` (509 satır) — Yapı Hazırlama

**Şelale (her yapı için):**

```
Ham PDB  →  clean_pdb()  →  extract_ligand_pdb()
                                    ↓
           RCSB CIF  →  cif_to_mol2() [CCD atom isimleri]
                                    ↓
                     molfile_to_params.py  →  .params
                                    ↓
                           PyRosetta pose yükle
```

**Yedekleme:** CIF başarısız olursa SDF(RDKit) → son çare PDB(obabel)

### `src/run_pipeline.py` (436 satır) — Ana Runner

- `--mode validate`: 1LYZ lizozim validasyonu
- `--mode single`: Tek yapı analizi
- `--mode all`: 25 EGFR → Pearson r + scatter plot

### `src/test_frustration.py` (267 satır) — 9 Birim Test

---

## 5. Çözülen Teknik Engeller

### 5.1 PyRosetta Kurulumu
**Sorun:** Wheel platform uyumsuzluğu.  
**Çözüm:** tar.bz2 → manuel extract → `pip install .` from `setup/`.

### 5.2 `rosetta_py` Eksikliği
**Sorun:** `molfile_to_params.py` `rosetta_py` modülünü gerektiriyor, PyRosetta paketinde yok.  
**Çözüm:** Saf Python modülü RosettaCommons/rosetta GitHub'dan indirildi → `src/rosetta_py/`.

### 5.3 OpenBabel mol2 Valans Hatası
**Sorun:** Aromatik atomlara `ar` bağ tipi → `assign_rosetta_types` crash.  
**Çözüm:** RDKit ile Kekulé formuna çevrilen SDF kullanıldı.

### 5.4 `fill_missing_atoms` Hatası (6 yapı)
**Sorun:** Params atom isimleri (`C1`, `C2`) ≠ PDB HETATM isimleri (`C16`, `C17`).  
**Çözüm:** RCSB CIF'ten `_chem_comp_atom.atom_id` ile mol2 üretildi → **25/25 pose yükleme başarılı**.

### 5.5 Çok Zincirli Yapılar
**Sorun:** HETATM filtresi zincir gözetmiyordu → birden fazla ligand kopyası dahil oluyordu.  
**Çözüm:** `chain == target_chain` koşulu eklendi.

---

## 6. Test Sonuçları

```
test_build_contact_partner_map_basic      ✅ PASSED
test_frustration_index_formula            ✅ PASSED
test_frustration_class_thresholds         ✅ PASSED
test_summarize_ligand_frustration_basic   ✅ PASSED
test_get_protein_contacts_count           ✅ PASSED
test_pairwise_energy_symmetric            ✅ PASSED
test_native_aa_frequency_sums_to_one      ✅ PASSED
test_decoy_backbone_unchanged             ❌ FAILED  (0.78 Å > 0.05 Å tolerans)
test_frustration_survey_small             ✅ PASSED

8/9 geçiyor
```

**Kalan hata:** PyRosetta 2026.25'te `FastRelax.set_movemap(mm)` + `mm.set_bb(False)` backbone'u tam donduramıyor. Düzeltme: `FastRelax` → `MinMover(chi-only)`.

---

## 7. Aşama Durumları

| Aşama | Durum |
|-------|-------|
| Aşama 0: PyRosetta kurulumu | ✅ Tamamlandı |
| Aşama 1: 25 PDB hazırlık (params + pose) | ✅ Tamamlandı — 25/25 |
| Aşama 2: Frustrasyon motoru (Eq.1, Eq.2) | ✅ Tamamlandı |
| Birim testler | ✅ 8/9 geçiyor |
| Aşama 3: Lizozim validasyonu | 🔲 Bekliyor |
| Aşama 4: EGFR analizi + korelasyon | 🔲 Bekliyor |

---

## 8. Sıradaki Adımlar

```bash
# 1. FastRelax → MinMover düzeltmesi (frustration.py ~5 satır)

# 2. Validasyon
python src/run_pipeline.py --mode validate --n_decoys 50

# 3. Tam analiz
python src/run_pipeline.py --mode all --n_decoys 200
```

---

*Bu proje, Chen et al. (2020)'nin atomistik frustrasyonunu EGFR–inhibitör komplekslerine uygulayan bağımsız bir reimplementasyondur. Orijinal yazarların kodu yayınlanmamıştır.*
